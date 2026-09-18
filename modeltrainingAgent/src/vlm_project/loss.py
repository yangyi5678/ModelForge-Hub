
class VLMLoss:

    def __call__(self, model, batch):
        outputs = model(
            **batch,
        )

        return outputs.loss


def build_loss(cfg):
    return VLMLoss()