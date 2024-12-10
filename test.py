import base64

# 이미지 파일을 base64로 변환
with open("image.jpg", "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')

# 변환된 문자열 출력 (이걸 복사해서 Postman에 붙여넣을 겁니다)
print(base64_string)