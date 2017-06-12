from creamas.core.artifact import Artifact

class MazeArtifact(Artifact):
    def __str__(self):
        return "Maze by:{} {}".format(self.creator,
                                            self.framings[self.creator])

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return str(self) < str(other)