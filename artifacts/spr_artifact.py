from creamas.core.artifact import Artifact

class SpiroArtifact(Artifact):
    '''Artifact class for Spirographs.
    '''
    def __str__(self):
        return "Spirograph by:{} {}".format(self.creator,
                                            self.framings[self.creator])

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash((self.creator, tuple(self.framings[self.creator]['args'])))