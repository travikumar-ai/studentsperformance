import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime("%d_%m_%y")}.log"

log_file_path = os.path.join(
                            os.getcwd(),
                            'logs',
                            )

os.makedirs(
            log_file_path,
            exist_ok=True
            )

LOG_FILE = os.path.join(log_file_path, 
                        LOG_FILE_NAME
                        )

format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
                    filename=LOG_FILE,
                    level=logging.INFO,
                    format= format
                    )

if __name__ == "__main__":
    logging.info('Log file testing started')
    logging.info('Log file testing finished')