class PerformanceMetrics:
    __train_loss: float
    __train_accuracy: float
    __valid_loss: float
    __valid_accuracy: float
    __test_loss: float
    __test_accuracy: float

    def __init__(self, train_loss: float, train_accuracy: float, valid_loss: float, valid_accuracy: float,
                 test_loss: float, test_accuracy: float) -> None:
        self.__train_loss = train_loss
        self.__train_accuracy = train_accuracy
        self.__valid_loss = valid_loss
        self.__valid_accuracy = valid_accuracy
        self.__test_loss = test_loss
        self.__test_accuracy = test_accuracy

    @property
    def train_loss(self) -> float:
        return self.__train_loss

    @property
    def train_accuracy(self) -> float:
        return self.__train_accuracy

    @property
    def valid_loss(self) -> float:
        return self.__valid_loss

    @property
    def valid_accuracy(self) -> float:
        return self.__valid_accuracy

    @property
    def test_loss(self) -> float:
        return self.__test_loss

    @property
    def test_accuracy(self) -> float:
        return self.__test_accuracy
