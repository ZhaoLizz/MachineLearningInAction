import requests

#
# url = 'http://127.0.0.1:8000/polls/'
# response = requests.get(url)
# print(response.text)
from django.urls import reverse

question_id = 1
# url = reverse('polls:vote', args=(question_id))


url = 'http://127.0.0.1:8000/polls/1/vote/'

session = requests.Session()
r = session.get(url)
session.headers.update({
    'Referer': url,
    'X-CSRFToken': r.cookies['csrftoken'],
})

params = {'choice': 4}

r = session.post(url,data=params)
print(r.status_code)

# response = requests.post(url,data=params)
# print(response.url)
