
import sys
import logging
from src.utills import logger


def ExceptionConfig(error, sys) -> str:
    _, _, exc_td = sys.exc_info()
    
    file_name = exc_td.tb_frame.f_code.co_filename
    line_no = exc_td.tb_lineno
    
    err_msg = ( f"Error occurred at file name [{file_name}]"
                f" at line no [{line_no}] error msg [{str(error)}])"
                )
    
    return err_msg


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = ExceptionConfig(
            error=error_message, sys=error_detail
        )

if __name__ == "__main__":
    logging.info("Starting test run that will handle an exception")
    try:
        a = 1/0
    except Exception as e:
        logging.error('Critical Error occurred and handled')
        error_msg = CustomException(e, sys)
        logging.error(f'Error details : {error_msg}')
        
    logging.info('Test run finished.')
