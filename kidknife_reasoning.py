class KidKnifeReasoning:

    def __init__(self):

        print("Kid-Knife Reasoning Ready")
        
    def analyze(self, state):

        if state == "SAFE":
            return False, "SAFE"

        if state == "DANGER":
            return True, "DANGER: Knife near child"

        if state == "EMERGENCY":
            return True, "EMERGENCY: Child holding knife!"

        return False, "UNKNOWN"

    # def analyze(self, knife, person):

    #     if knife and person:

    #         return True, "DANGER: Child near knife detected"

    #     return False, "Safe"