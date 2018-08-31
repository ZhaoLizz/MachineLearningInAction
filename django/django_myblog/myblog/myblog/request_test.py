import requests

url = 'http://127.0.0.1:8000/blog/article/1'
r = requests.get(url)

print(r.content)