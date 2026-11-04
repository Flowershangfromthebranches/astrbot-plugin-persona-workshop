from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

from astrbot import logger

from .consultation import SilentConsultation
from .livingmemory_bridge import LivingMemoryBridge


class RoomRuntimeManager:
    """Run one hidden continuum Room session per AstrBot conversation/persona."""

    def __init__(
        self,
        runtime: Any,
        context: Any,
        config: Any,
        links_loader: Callable[[], dict[str, str]],
        data_dir: Path,
    ) -> None:
        self.runtime = runtime
        self.context = context
        self.config = config
        self.links_loader = links_loader
        self.livingmemory = LivingMemoryBridge(context)
        self.consultation = SilentConsultation(context, config)
        self.sessions_path = Path(data_dir) / "runtime_sessions.json"
        self._sessions = self._load_sessions()

        from persona_continuum.room.models import ParticipantSlot
        from persona_continuum.room.prompt_composer import PromptComposer

        self.ParticipantSlot = ParticipantSlot
        self.composer = PromptComposer()

    def _load_sessions(self) -> dict[str, str]:
        try:
            data = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(key): str(value) for key, value in data.items() if value}

    def _save_sessions(self) -> None:
        temp = self.sessions_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self._sessions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(self.sessions_path)

    def _native_to_continuum(self, native_persona_id: str) -> str:
        links = self.links_loader()
        for continuum_id, native_id in links.items():
            if native_id == native_persona_id:
                return continuum_id
        return ""

    def _memory_mode(self) -> str:
        value = str(self.config.get("memory_mode", "auto") or "auto").strip().lower()
        if value not in {"auto", "continuum", "livingmemory", "hybrid"}:
            return "auto"
        return value

    def _room_id(self, continuum_persona_id: str, umo: str) -> str:
        key = f"{continuum_persona_id}|{umo}"
        room_id = self._sessions.get(key, "")
        if room_id:
            try:
                self.runtime.continuum.rooms.get_state(room_id)
                return room_id
            except Exception:
                logger.warning("[persona_workshop] Room 不存在，重新创建：%s", room_id)
                self._sessions.pop(key, None)
        room = self.runtime.continuum.rooms.create_room(
            [continuum_persona_id],
            topic=f"AstrBot:{umo}",
        )
        room_id = str(room["id"])
        self._sessions[key] = room_id
        self._save_sessions()
        return room_id

    async def prepare(self, native_persona_id: str, umo: str, user_message: str) -> dict[str, Any]:
        continuum_persona_id = self._native_to_continuum(native_persona_id)
        if not continuum_persona_id:
            raise RuntimeError("persona_not_linked_to_continuum")
        room_id = self._room_id(continuum_persona_id, umo)
        bundle = await asyncio.to_thread(
            self.runtime.continuum.rooms.prepare_next,
            room_id,
            user_message,
        )
        prepared = bundle["prepared"]
        room_state = bundle["room"]
        session_id = bundle["session_id"]
        persona_id = bundle["speaker_persona_id"]
        manifest = prepared.identity_anchor
        original_prepared = prepared.model_copy(deep=True)
        slot = self.ParticipantSlot(
            participant_id=persona_id,
            persona_id=persona_id,
            role="member",
            display_name=str(manifest.display_name or persona_id),
            tool_permissions=[],
        )

        memory_mode = self._memory_mode()
        living_available = self.livingmemory.available
        inject_living = (
            living_available
            and (
                memory_mode in {"livingmemory", "hybrid"}
                or bool(self.config.get("livingmemory_inject", False))
            )
        )
        if living_available and memory_mode in {"auto", "livingmemory"}:
            # LivingMemory's own on_llm_request hook injects retrieved memories
            # as temporary user content.  Do not put the same memory in the
            # system prompt again; only the persona/state layers come from Room.
            prepared = prepared.model_copy(
                update={
                    "relevant_memories": [],
                    "activated_emotional_memories": [],
                }
            )

        system_prompt, _, _ = self.composer.compose_turn_prompt(
            slot,
            prepared=prepared,
            dynamic_recall_memories=[],
            recent_transcript=[],
            room_topic=room_state.get("topic"),
            user_message=user_message,
            context_mode="delta",
        )
        system_prompt = re.sub(r"(?m)^# ", "## ", system_prompt).strip()

        living_items: list[dict[str, Any]] = []
        if inject_living:
            living_items = await self.livingmemory.search(
                user_message,
                session_id=umo,
                k=max(1, min(20, int(self.config.get("livingmemory_top_k", 5) or 5))),
            )
            block = LivingMemoryBridge.format_block(living_items)
            if block:
                system_prompt += "\n\n" + block

        if self.consultation.should_run(user_message):
            try:
                memory_snippets = list(original_prepared.relevant_persona_facts or [])
                memory_snippets.extend(
                    str(item.content)
                    for item in (original_prepared.relevant_memories or [])
                    if getattr(item, "content", None)
                )
                memory_snippets.extend(str(item.get("content") or "") for item in living_items)
                state_parts = [
                    "情绪："
                    + "、".join(
                        f"{item.name}={item.intensity:.2f}"
                        for item in original_prepared.current_emotions
                        if item.intensity > 0.05
                    ),
                    "需求："
                    + "、".join(
                        f"{item.name}={item.level:.2f}"
                        for item in original_prepared.current_needs
                        if item.level > 0.05
                    ),
                    "关系立场：" + str(original_prepared.relationship_stance or ""),
                    "目标：" + "、".join(str(goal) for goal in original_prepared.active_goals or []),
                ]
                brief = await self.consultation.run(
                    persona_name=str(manifest.display_name or persona_id),
                    system_prompt=system_prompt,
                    user_message=user_message,
                    state_text="\n".join(part for part in state_parts if part),
                    memories=[item for item in memory_snippets if item],
                )
                if brief:
                    system_prompt += (
                        "\n\n## 静默会诊备忘（仅供你内部参考，绝对不要向用户提及会诊过程）\n"
                        + brief
                    )
                    logger.info(
                        "[persona_workshop] 静默会诊完成：brief=%s字",
                        len(brief),
                    )
            except Exception:
                logger.exception("[persona_workshop] 静默会诊失败，继续普通 Room 回复")

        return {
            "room_id": room_id,
            "session_id": session_id,
            "persona_id": persona_id,
            "native_persona_id": native_persona_id,
            "umo": umo,
            "user_message": user_message,
            "system_prompt": system_prompt,
            "memory_backend": (
                "livingmemory"
                if living_available and memory_mode in {"auto", "livingmemory"}
                else ("hybrid" if living_available and memory_mode == "hybrid" else "continuum")
            ),
            "livingmemory_count": len(living_items),
        }

    def commit(self, turn: dict[str, Any], assistant_response: str) -> None:
        if not assistant_response.strip():
            return
        self.runtime.continuum.rooms.commit_turn(
            turn["room_id"],
            turn["persona_id"],
            turn["session_id"],
            turn["user_message"],
            assistant_response,
        )
