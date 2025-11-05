from datetime import datetime
from dateutil.relativedelta import relativedelta

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    
    # Convert the date string to a datetime object
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Generate a list of first-of-month dates
    first_of_month_dates = []
    
    current_date = start_date
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += relativedelta(months=1)
    
    return first_of_month_dates