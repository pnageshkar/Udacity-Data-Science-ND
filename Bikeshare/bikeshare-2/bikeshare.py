# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:31:51 2018

@author: Prasad Nageshkar
"""

import time
import pandas as pd
#import numpy as np

CITY_DATA = { 'Chicago': 'chicago.csv',
              'New York City': 'new_york_city.csv',
              'Washington': 'washington.csv' }
month_list  = ['January','February','March','April','May','June','All']
day_list    = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','All']
city_list   = ['Chicago','New York City','Washington']

def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    
    filter_list = ['B','M','D','N']
    print('*** Welcome to exploration of US bikeshare data ***')
    print("Note : We currently offer Bikeshare information on only Chicago,New York City and Washington")
    # get user input for city (chicago, new york city, washington)
    city_sel = False
    while not city_sel:
        city = str(input("1> Please enter the name of the city that you would like to explore : "))
        city = city.title()
        if city in city_list:
            city_sel = True
        else:
            print("Your entry of " + city + " is not valid , Try again !")
        
    filter_sel = False
    while not filter_sel:
        print("2> How do you want to filter the data by ?")
        filter_choice = str(input("Type 'M' for Month,'D' for Day of week, 'B' for both or 'N' for No filter : "))
        filter_choice = filter_choice.upper()
        if filter_choice in filter_list:
            filter_sel = True
        else:
             print("Your entry of " + filter_choice + " is not valid , Try again !")
    if filter_choice == 'B': # Both Month and Day filter
        month_sel = False
        while not month_sel:
            month = str(input("3> Enter a month  from the following : January,February,March,April,May,June or 'All' for All months : "))
            month = month.title()
            if month in month_list:
                month_sel = True
            else:
                print("Your entry of " + month + "is invalid")
        day_sel = False
        while not day_sel:
            day = str(input ("4> Enter a day of the week  e.g.: Monday ... Sunday or 'All' for All days : "))
            day = day.title()
            if day in day_list:
                day_sel = True
            else:
                print("Your entry of " + day + "is invalid")
    elif filter_choice == 'M': # Filter by Month
        month_sel = False
        while not month_sel:
            month = str(input("3> Enter a month  from the following : January,February,March,April,May,June or 'All' for All months : "))
            month = month.title()
            if month in month_list:
                month_sel = True
            else:
                print("Your entry of " + month + "is invalid")
        day = 'All'
    elif filter_choice == 'D': # Filter by day
        month = 'All'
        day_sel = False
        while not day_sel:
            day = str(input ("3> Enter a day of the week  e.g.: Monday ... Sunday or 'All' for All days : "))
            day = day.title()
            if day in day_list:
                day_sel = True
            else:
                print("Your entry of " + day + "is invalid")
    elif filter_choice == 'N': # No filter
        month = 'All'
        day   = 'All'
   
    print('-'*80)
    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        dfBikeshare - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    #start_time = time.time()
    #print("Loading data ......")
    dfBikeshare = pd.read_csv(CITY_DATA[city])
     # convert the Start Time column to datetime
    dfBikeshare['Start Time'] = pd.to_datetime(dfBikeshare['Start Time'])
    # extract month and day of week from Start Time to create new columns
    dfBikeshare['month'] = dfBikeshare['Start Time'].dt.month
    dfBikeshare['day_of_week'] = dfBikeshare['Start Time'].dt.weekday_name
    
    # filter by month if applicable
    if month != 'All':
        # use the index of the months list to get the corresponding int
        months = ['January', 'February', 'March', 'April', 'May', 'June']
        month = months.index(month) + 1
        # filter by month to create the new dataframe
        dfBikeshare = dfBikeshare[dfBikeshare['month'] == month]
    # filter by day of week if applicable
    if day != 'All':
        # filter by day of week to create a new dataframe
        dfBikeshare = dfBikeshare[dfBikeshare['day_of_week'] == day]
        
    #print("\nLoading data took {:.6} seconds." .format(time.time() - start_time))         
    return dfBikeshare
    


def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    #start_time = time.time()

    # display the most common month
    month_index = df['month'].mode()[0] - 1
    print("a) The most frequently travelled month : {} ".format(month_list[month_index]))

    # display the most common day of week
    print("b) The most frequently travelled day : {} ".format(df['day_of_week'].mode()[0]))

    # display the most common start hour
    print("c) The most frequent Start hour : {}:00 ".format(df['Start Time'].dt.hour.mode()[0]))

    #print("\nThis took {:.6} seconds " .format(time.time() - start_time))
    print('-'*80)


def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    #start_time = time.time()

    # display most commonly used start station
    print("a) The most commonly used Start Station : {} ".format(df['Start Station'].mode()[0]))

    # display most commonly used end station
    print("b) The most commonly used End Station : {}".format(df['End Station'].mode()[0]))

    # display most frequent combination of start station and end station trip
    dfgroup= df.groupby(['Start Station', 'End Station']).size().reset_index(name='Trip Counts')
    dfsorted = dfgroup.sort_values('Trip Counts', ascending=False)
    
    print("c) The most common trip is between : {} & {} ".format(dfsorted.iloc[0,0] , dfsorted.iloc[0,1]))

    #print("\nThis took {:.6} seconds." .format(time.time() - start_time))
    print('-'*80)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    #start_time = time.time()

    # display total travel time in hours (convert from seconds)
    total_trip_time = df['Trip Duration'].sum()/360
    print("a) Total Trip time =  {:.0f} hours ".format(total_trip_time))
    # display Total number of trips
    print("b) Total number of Trips = {}".format(df['Trip Duration'].count())) 
    # display mean travel time in minutes (convert from seconds)
    avg_travel_time = df['Trip Duration'].mean() / 60
    print("c) Average Trip time = {:.2f} minutes".format(avg_travel_time))
    
    #print("\nThis took {:.6} seconds." .format(time.time() - start_time))
    print('-'*80)


def user_stats(df,city):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    #start_time = time.time()

    # Display counts of user types
    customer_count = df[df['User Type'] == 'Customer'].count()[0]
    subscriber_count = df[df['User Type'] == 'Subscriber'].count()[0]
    print("a) Count of Customers = {}".format(customer_count))
    print("b) Count of Subscribers = {}".format(subscriber_count))

     
    if city == 'New York City' or city == 'Chicago':
        # Display counts of gender
        male_count = df[df['Gender'] == 'Male'].count()[0]
        female_count = df[df['Gender'] == 'Female'].count()[0]
        print("c) Count of Male Users of the Service = {}".format(male_count))
        print("d) Count of Female Users of the Service = {}".format(female_count))

        # Display earliest, most recent, and most common year of birth
        birthyear_oldest = df['Birth Year'].min()
        birthyear_youngest = df['Birth Year'].max()
        birthyear_common = df['Birth Year'].mode()[0]
        print("e) Birth Year of Oldes User = {}".format(int(birthyear_oldest)))
        print("f) Birth Year of Youngest User = {}".format(int(birthyear_youngest)))
        print("g) Most common Birth Year amongst Users = {}".format(int(birthyear_common)))
    else:
        print ('Gender and Birth Year data not available !')
    #print("\nThis took {:.6} seconds." .format(time.time() - start_time))
    print('-'*80)

def display_data(df):
    """ Displays Raw data to the user - 5 lines at a time till the user Stops """
    
    dis_choice = str(input("Do you want to view the raw data ? Enter Y to continue OR any other key to skip !")).upper()
    if dis_choice == 'Y':
        total_row_count = df.shape[0] # Total number of rows in the dataframe
        #print("Total Rows " , total_row_count)
        # The number of rows that will be displayed for the user to scroll at a time
        maxrows_per_page = 5 # you can change this number if more rows need to be displayed
        # Number of Pages with  rows/page = maxrows_per_page
        page_count = total_row_count/maxrows_per_page 
        #print("page Count : ", page_count)
        # When remaining number of rows are < max rows per page
        rows_lt_maxrows_page = total_row_count % maxrows_per_page 
        #print("Balance rows : ", rows_lt_maxrows_page)
        if total_row_count == 0:
            print("No Data to display !")
        elif page_count == 0:
            print("Displaying the Data below :\n")
            # if the total number of rows are less than maxrows/page
            print(df.iloc[0:rows_lt_maxrows_page])
        elif page_count > 0:
            page_index = 1
            row_lower_index = 0
            row_upper_index = maxrows_per_page
            show_raw_data = True
            # First display all pages with max rows/page
            while show_raw_data and page_index <= page_count:
                print(df.iloc[row_lower_index : row_upper_index])
                nextset =  str(input("\nDo you want to see more rows ? Enter 'S' to stop and any other key to continue !")).upper()
                if nextset == 'S':
                    show_raw_data = False
                else:
                    row_lower_index = row_upper_index
                    row_upper_index += maxrows_per_page
                    page_index += page_index + 1
            if show_raw_data == True and row_upper_index < total_row_count:
                # print the pending rows on last page (the count of which is less than max rows/page)
                print(df.iloc[row_upper_index :total_row_count])
    else:
        print("Skipping the display of raw data as desired by you !!")
        
            
def main():
   
    #Execution of code starts here
    app_run = True
    while app_run:
        # Get the elements from the user for the filtering the data / extracting information
        city, month, day = get_filters()
        # load data based on the city , month and day selected
        dfBikeshare_filter = load_data(city, month, day)
        # Determine Popular (Frequent) Times for Travel
        time_stats(dfBikeshare_filter)
        # Determine commonly used Stations
        station_stats(dfBikeshare_filter)
        # Calculate Trip duration - Total, Average
        trip_duration_stats(dfBikeshare_filter)
        # Determine User stats
        user_stats(dfBikeshare_filter,city)
        # display raw data
        display_data(dfBikeshare_filter)
        # Check if the user wants to Continue or Exit
        exit_prog = str(input("\nWould you like to Continue ? Enter 'Y' to Continue or any other key to Exit \n")).upper()
        if exit_prog != 'Y':
            app_run = False

if __name__ == "__main__":
	main()
