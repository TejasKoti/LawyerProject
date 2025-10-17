# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import csv

# Base URL of the website to scrape
base_url = 'https://kanoongurus.com/search?lawyerName=&city=&state=&page='
total_pages = 72  # adjust if site changes

with open('Extracted_Lawyers.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Experience', 'Specialization', 'Location', 'Rating', 'Profile URL'])

    for page in range(1, total_pages + 1):
        url = base_url + str(page)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        lawyer_blocks = soup.find_all('div', class_='bs-box')

        for block in lawyer_blocks:
            name = block.find('div', class_='custom-value-name').text.strip()
            experience = block.find('div', class_='custom-label', text='Experience').find_next_sibling('div').text.strip()
            specialization = block.find('div', class_='custom-label', text='Specialization').find_next_sibling('div').text.strip()
            location = block.find('div', class_='custom-label', text='Location').find_next_sibling('div').text.strip()
            rating_block = block.find('div', class_='rating-tag')
            rating = rating_block.text.strip() if rating_block else 'No rating'
            profile_url = f"https://kanoongurus.com{block.find('a')['href']}"

            writer.writerow([name, experience, specialization, location, rating, profile_url])

        print(f"Page {page} done.")

print("✅ Data extraction completed! Saved as 'Extracted_Lawyers.csv'")

import pandas as pd
import re

df = pd.read_csv("Extracted_Lawyers.csv")
print(df.shape)
df.head()

# Extract numeric values from "Experience"
df['Experience_Years'] = df['Experience'].apply(lambda x: int(re.search(r'\d+', str(x)).group()) if pd.notnull(x) else 0)

# Fill missing values
df.fillna("Unknown", inplace=True)

df.to_csv("Cleaned_Lawyers.csv", index=False)
print("✅ Cleaned dataset saved as 'Cleaned_Lawyers.csv'")
df.head()

def assign_ratings(years):
    if years >= 20: return 5.0
    elif years >= 15: return 4.5
    elif years >= 10: return 4.0
    elif years >= 5: return 3.0
    else: return 2.0

df['Generated_Rating'] = df['Experience_Years'].apply(assign_ratings)
df.to_csv("Processed_Lawyers.csv", index=False)

print("✅ Ratings generated and saved to 'Processed_Lawyers.csv'")
df[['Name', 'Experience_Years', 'Generated_Rating']].head()

def recommend_lawyers(city=None, min_experience=0, specialization=None):
    result = df.copy()
    if city:
        result = result[result['Location'].str.contains(city, case=False, na=False)]
    if specialization:
        result = result[result['Specialization'].str.contains(specialization, case=False, na=False)]
    result = result[result['Experience_Years'] >= min_experience]
    return result[['Name', 'Experience_Years', 'Specialization', 'Location', 'Generated_Rating', 'Profile URL']]

# Example
recommend_lawyers(city="Mumbai", min_experience=10, specialization="Criminal")