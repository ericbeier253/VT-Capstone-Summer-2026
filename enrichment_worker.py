import queue
import threading

class AsyncEnrichmentWorker:

    def __init__(
        self,
        storage_handler,
        analyzer,
        cropper,
        embedder,
        matcher,
        repository,
        logger=None,
    ):
        self.storage_handler = storage_handler
        self.analyzer = analyzer
        self.cropper = cropper
        self.embedder = embedder
        self.matcher = matcher
        self.repository = repository
        self.logger = logger
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def start(self):
        self.thread.start()

    def enqueue(self, row_obj, run_id):
        self.queue.put((row_obj, run_id))

    def process_image(
        self,
        row_obj,
        run_id,
    ):

        try:

            analysis = self.analyzer.analyze(
                row_obj.img_path
            )

            crops = self.cropper.crop_objects(
                row_obj.img_path,
                analysis,
            )

            if not crops:
                return

            crop_paths = [crop.crop_path for crop in crops]
            embeddings = self.embedder.embed_batch(crop_paths)

            for crop, embedding in zip(crops, embeddings):

                object_id = self.matcher.assign_object_id(
                    embedding
                )

                self.repository.save_object(

                    object_id=object_id,

                    embedding=embedding,

                    crop=crop,

                    run_id=run_id,

                    parent_image=row_obj.img_path,

                )

            log_str = self.storage_handler.save_event(

                timestamp=row_obj.timestamp,

                depth=row_obj.depth,

                img_path=row_obj.img_path,

                run_id=run_id,

                llm_analysis=analysis.model_dump(),

            )
            if log_str and self.logger:
                self.logger.info(log_str.strip())
            elif log_str:
                print(log_str.strip())

            if self.logger:
                self.logger.info(
                    "Processed %s (%d objects)",
                    row_obj.img_path,
                    len(crops),
                )

        except Exception:

            if self.logger:
                self.logger.exception(
                    "Failed processing %s",
                    row_obj.img_path,
                )

            raise

    def worker(self):

        while True:

            item = self.queue.get()

            if item is None:
                break

            row_obj, run_id = item

            try:
                self.process_image(row_obj, run_id)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Worker thread caught exception processing {row_obj.img_path}: {e}")
                else:
                    print(f"Worker thread caught exception: {e}")
            finally:
                self.queue.task_done()


