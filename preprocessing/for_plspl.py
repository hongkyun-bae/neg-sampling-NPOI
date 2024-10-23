import csv
from datetime import datetime

input_file_path = "./input/Bri_plspl.txt"  # 원본 파일 경로
output_file_path = "./input/bri_small.txt" # 출력 파일 경로

# 원하는 출력 열
output_columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"]

with open(input_file_path, "r") as infile, open(output_file_path, "w", newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile, delimiter='\t')

    # 헤더 쓰기
    writer.writerow(output_columns)

    for row in reader:
        # 원하는 정보 가져오기
        userid = row["userid"]
        venueid = row["venueid"]
        catid = row["catid"]
        latitute = row["latitute"]
        longitude = row["longitude"]
        time = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%SZ").strftime("%a %b %d %H:%M:%S +0000 %Y")

        # catname과 timezone은 빈 문자열로 설정
        catname = " "
        timezone = " "

        # 새로운 형식으로 행 쓰기
        writer.writerow([userid, venueid, catid, catname, latitute, longitude, timezone, time])

print(f"{input_file_path} 파일의 형식이 변경되어 {output_file_path}에 저장되었습니다.")
