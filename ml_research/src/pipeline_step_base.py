from abc import ABC, abstractmethod

class PipleLineStepBase(ABC):
    
    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement the run method.")