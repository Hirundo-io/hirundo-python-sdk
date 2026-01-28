from enum import Enum


class BiasType(str, Enum):
    ALL = "ALL"
    RACE = "RACE"
    NATIONALITY = "NATIONALITY"
    GENDER = "GENDER"
    PHYSICAL_APPEARANCE = "PHYSICAL_APPEARANCE"
    RELIGION = "RELIGION"
    AGE = "AGE"
