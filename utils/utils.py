class PRScore:
    def __init__(self, threshold=0.5):
        super().__init__()
        self.num_true = 0
        self.num_pred = 0
        self.num_target = 0
        self.threshold = threshold

    def calc(self):
        # summarize the metrics
        res_info = {
            "precision": self.num_true / (self.num_pred + 1e-6),
            "recall": self.num_true / (self.num_target + 1e-6)
        }
        return res_info

    def reset(self):
        # the metric will reset after summary
        self.num_true = 0
        self.num_pred = 0
        self.num_target = 0

    def collect(self, labels, preds):
        """
        :param labels: ground truth label
        :param preds: predictions
        :return:
        """
        if isinstance(labels, list):
            assert len(labels) == 1
            labels = labels[0]
        logits = preds
        labels = labels.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        assert len(labels) == len(logits)
        pred = logits > self.threshold
        self.num_true += ((labels == pred) * labels).sum()
        self.num_pred += pred.sum()
        self.num_target += labels.sum()

    @staticmethod
    def name():
        return "Precison"