"""
Useful Functions.

Authors: Hongjie Fang.
"""

def display_results(metrics_dict, logger):
    """
    Given a metrics dict, display the results using the logger.

    Parameters
    ----------
        
    metrics_dict: dict, required, the given metrics dict;

    logger: logging.Logger object, the logger.
    """
    try:
        display_list = []
        for key in metrics_dict.keys():
            if key == 'samples':
                num_samples = metrics_dict[key]
            else:
                display_list.append([key, float(metrics_dict[key])])
        logger.info("Metrics on {} samples:".format(num_samples))
        for display_line in display_list:
            metric_name, metric_value = display_line
            logger.info("  {}: {:.6f}".format(metric_name, metric_value))    
    except Exception:
        logger.warning("Unable to display the results, the operation is ignored.")
        pass