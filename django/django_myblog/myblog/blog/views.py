from django.shortcuts import render
from django.http import HttpResponse

from . import models


def index(request):
    articles = models.Article.objects.all()  # 从数据库中查询获取model对象
    return render(
        request,
        'blog/index.html',
        {'articles': articles}
    )


def article_page(request, article_id):
    article = models.Article.objects.get(pk=article_id)
    return render(request, "blog/article_page.html", {'article': article})


def edit_page(request, article_id):
    # 如果是新文章,就打开编辑新文章页面 (0是在index.html的超链接里传入的固定值)
    if str(article_id) == '0':
        return render(request, 'blog/edit_page.html')
    # 修改文章
    article = models.Article.objects.get(pk=article_id)
    return render(request, 'blog/edit_page.html', {'article': article})


def edit_action(request):
    # 传送的是POST数据,从request的POST中获取
    title = request.POST.get('title', 'DEFAULT_TITLE')
    content = request.POST.get('content', 'DEFAULT_CONTENT')
    article_id = request.POST.get('article_id', '0')

    # 如果是新建一个博客
    if (article_id == '0'):
        models.Article.objects.create(title=title, content=content)
        # 最后返回主页
        articles = models.Article.objects.all()  # 从数据库中查询获取model对象
        return render(
            request,
            'blog/index.html',
            {'articles': articles}
        )
    # 如果是修改已有的博客
    else:
        article = models.Article.objects.get(pk=article_id)
        article.title = title
        article.content = content
        article.save()
        return render(request, 'blog/article_page.html', {'article': article})
