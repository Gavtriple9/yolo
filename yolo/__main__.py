#!/usr/bin/env python3
import sys

from yolo.parser import get_parser
from yolo.logger import get_root_logger, ROOT_LOGGER_NAME
import yolo.functional as fn


def main():
    parser = get_parser()
    args = parser.parse_args()
    logger = get_root_logger(ROOT_LOGGER_NAME, args.log_level, args.verbose)

    if args.prog == "train":
        logger.info("Training...")
        fn.train_model(args, logger)

    elif args.prog == "test":
        logger.info("Testing...")
        fn.test_model(args, logger)

    else:
        logger.error(f"Unknown program: {args.prog}")
        sys.exit(1)

    logger.info("Yolo finished successfully.")


if __name__ == "__main__":
    main()
