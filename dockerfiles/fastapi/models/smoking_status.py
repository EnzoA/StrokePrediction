from enum import Enum

class SmokingStatus(Enum):
    never_smoked = 'never smoked'
    unknown = 'Unknown'
    formerly_smoked = 'formerly smoked'
    smokes = 'smokes'