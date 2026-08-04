from abc import ABC, abstractmethod

from .structures import AnnotationRecord, AnnotationRequest


class BaseAnnotator(ABC):

    @abstractmethod
    def annotate(self, request: AnnotationRequest) -> AnnotationRecord:
        """Produce one typed record with provenance and confidence."""
