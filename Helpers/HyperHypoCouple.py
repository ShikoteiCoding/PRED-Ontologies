class HHCouple:
    def __init__(self, hypo, hyper):
        self.hypernym = hyper
        self.hyponym = hypo

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(" + self.hyponym + ", " + self.hypernym + ")"

    def __eq__(self, other):
        return self.hypernym == other.hypernym and self.hyponym == other.hyponym

    def __ne__(self, other):
        """Override the default Unequal behavior"""
        return self.hypernym != other.hypernym or self.hyponym != other.hyponym

    def __hash__(self):
        return hash(str(self))


class NHHCouple:
    def __init__(self, nhypo, nhyper):
        self.nhyper = nhyper
        self.nhypo = nhypo

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "(" + self.nhypo + ", " + self.nhyper + ")"

    def __eq__(self, other):
        return self.nhyper == other.nhyper and self.nhypo == other.nhypo

    def __ne__(self, other):
        """Override the default Unequal behavior"""
        return self.nhyper != other.nhyper or self.nhypo != other.nhypo

    def __hash__(self):
        return hash(str(self))