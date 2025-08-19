from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ConversationType(Enum):
    SIMPLE = "simple"
    TEACHER_STUDENT = "teacher_student"


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TeacherStudentConversation:
    conversation_type: ConversationType
    messages: List[ConversationMessage]
    student_image_count: int
    teacher_image_counts: Optional[List[int]] = None


__all__ = [
    "ConversationType",
    "ConversationMessage",
    "TeacherStudentConversation",
]
