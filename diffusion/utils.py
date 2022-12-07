from composer import Callback, Event, Logger, State, Time
from composer.callbacks.image_visualizer import _make_input_images
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger

class LogDiffusionImages(Callback):

    def __init__(self, interval='1000ba'):
        self.interval = Time.from_timestring(interval)

    def run_event(self, event: Event, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value
        if event == Event.BATCH_END and current_time_value % self.interval.value == 0:
            images = state.model.loop_p_sample()
            table = _make_input_images(images, 64)
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    destination.log_metrics({'Image': table}, state.timestamp.batch.value)

