from df_logger.logger import Logger

main_logger = Logger("Main logger").get_logger()

time_logger = Logger("Time logger", log_file="time.log").get_logger()
