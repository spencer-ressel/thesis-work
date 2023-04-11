#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# month_distance.py                                                           #                                                #
# Spencer Ressel                                                              #
# 2021.12.17                                                                  #   
###############################################################################

'''
This script calculates the distance between two dates in months. 

Input:        The two dates of interest.
Output:       The distance between them.
Figures:      None
Dependencies: None
'''

#### Month Distance calculator
def month_distance(start_year, start_month, end_year, end_month):
    '''
    Calculates the distance in number of months between two dates. The function
    will return a value of 0 if the two dates provided are the same. 

    Parameters
    ----------
    start_year : int
        The starting year.
    start_month : int
        The starting month, given as a numeric value from 1 to 12.
    end_year : int
        The ending year.
    end_month : int
        The ending month, given as a numeric value from 1 to 12.

    Returns
    -------
    int
        The number of months between the two dates provided. The same date 
        entered twice will give an output of 0

    '''
    # Calculate the number of years between the two year dates
    years = end_year - start_year - 1
    
    # Calcualte the number of months between the two month dates
    if start_month < end_month:
        months = (end_month - start_month) + 13
    else:
        months = 12 - (start_month - end_month) + 1
        
    # Calculate and return the total distance between the two dates, in months
    return 12*years + months - 1
