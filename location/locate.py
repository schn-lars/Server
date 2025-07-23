db_config = {
    # port, user and stuff
}

import re
import csv
import psycopg2
import unicodedata

def integrate():
    FILE = "./adresses_new.csv"
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    batch = []
    query = '''
        INSERT INTO adresses (street, number, zip, zip_label, name, canton, coord_x, coord_y) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    '''
    batch_size = 5000
    with open(FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        print(reader.fieldnames) 
        for row in reader:
            zip_label = row['ZIP_LABEL']
            zip = re.search(r'\b\d{4}\b', zip_label)
            if not zip:
                continue
            zip_code = zip.group()
            batch.append((
                row['STN_LABEL'],
                row['ADR_NUMBER'],
                zip_code,
                zip_label,
                row['COM_NAME'],
                row['COM_CANTON'],
                row['COORDX'],
                row['COORDY']
            ))
            if len(batch) >= batch_size:
                cursor.executemany(query, batch)
                batch.clear()
        if batch:
            cursor.executemany(query, batch)
    connection.commit()

    cursor.close()
    connection.close()

def normalize_street_name(street):
    if not street:
        return None
    street = street.lower()
    normalized = ''.join(
        c for c in street if unicodedata.category(c).startswith('L')
    )
    return normalized

def add_normalized_street():
    print("Starting with normalization")
    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    cursor.execute("ALTER TABLE adresses ADD COLUMN normalized_street TEXT;")
    connection.commit()

    fetch_query = "SELECT id, street FROM adresses;"
    cursor.execute(fetch_query)
    rows = cursor.fetchall()

    for row in rows:
        row_id, street = row
        normalized = normalize_street_name(street)
        update_query = "UPDATE adresses SET normalized_street = %s WHERE id = %s;"
        cursor.execute(update_query, (normalized, row_id))

    connection.commit()

    cursor.close()
    connection.close()
    print("Finished with normalization")

if __name__ == "__main__":
    integrate()
    add_normalized_street()




