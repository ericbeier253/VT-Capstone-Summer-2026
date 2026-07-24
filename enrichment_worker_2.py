

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

            for crop in crops:

                embedding = self.embedder.embed(
                    crop.crop_path
                )

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

            self.storage_handler.save_event(

                timestamp=row_obj.timestamp,

                depth=row_obj.depth,

                img_path=row_obj.img_path,

                run_id=run_id,

                metadata=analysis.model_dump(),

            )

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
                self.process_image(
                    row_obj,
                    run_id,
                )

            finally:
                self.queue.task_done()


