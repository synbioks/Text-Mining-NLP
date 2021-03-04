import os
import json 
from datetime import datetime
def extract_pub_year(input_path, output_path):
    json_years = {}
    walk_list = os.walk(input_path)
    for i in walk_list:
        root = i[0]
        for j in i[2]:
            p = root + '/' + j
            j_file = json.load(open(p))

            # only keep research articles 
            if j_file['is_research']:
                
                # first check electron_pub_data
                if len(j_file['electron_pub_date']) > 1:
                    json_years[j.replace('.json', '')] = datetime.strptime(j_file['electron_pub_date'], "%m/%d/%Y").year
                    
                # then check issue_pub_date
                elif len(j_file['issue_pub_date']) > 1:
                    json_years[j.replace('.json', '')] = datetime.strptime(j_file['issue_pub_date'], "%m/%d/%Y").year
                    
                # last check history 
                else:
                    print(j)
                    json_years[j.replace('.json', '')] = datetime.strptime(j_file['history'][0]["time"], "%m/%d/%Y").year
    with open(output_path, 'w') as f:
        json.dump(json_years, f)