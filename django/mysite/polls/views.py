from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.utils import timezone
from django.views import generic

from polls.models import Question, Choice


#
# def index(request):
#     latest_question_list = Question.objects.order_by('-pub_date')
#     # template = loader.get_template('polls/index.html')
#     context = {
#         'latest_question_list': latest_question_list
#     }
#     return render(request, 'polls/index.html', context)

class IndexView(generic.ListView):
    # 所有Question的列表
    template_name = 'polls/index.html'  # 通用视图对应的模板
    context_object_name = 'latest_question_list'  # 指定context变量名,用于在HTML中引用.

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        # return Question.objects.order_by('-pub_date')
        return Question.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by('-pub_date')



# def detail(request, question_id):
#     # 快捷函数
#     question = get_object_or_404(Question, pk=question_id)
#     return render(request, 'polls/detail.html', {'question': question})

class DetailView(generic.DetailView):
    """
    每个通用视图需要知道它将作用于哪个模型。 这由 model 属性提供。
    generic.DetailView期望从URL中捕获名为'pk'的主键值,所以我们为了通用视图,在urls里面把question_id改成pk
    """
    model = Question
    template_name = 'polls/detail.html'  # 手动指定模板的名字,否则就使用默认模板

    def get_queryset(self):
        """
        exclude any questions that aren't published yet
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


# def results(request, question_id):
#     question = get_object_or_404(Question,pk=question_id)
#     return render(request,'polls/results.html',{'question':question})

class ResultView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
    print('log',question_id)
    question = get_object_or_404(Question, pk=question_id)
    try:
        # 通过HTML里的form的input的name获取表单数据的value字段,这里可以获取到choice的ID
        select_choice = question.choice_set.get(pk=request.POST['choice'])
    except(KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': 'You did not select a choice'
        })
    else:
        # 如果没有发生异常
        select_choice.votes += 1
        select_choice.save()
        # always return an HttpResponseRedirect after successfully dealing
        # with POST data.This prevents data from being posted twice if a user
        # hits the Back button
        # 接收一个用户将要被重定向的URL
        # reverse()函数避免了我们这里 硬编码 URL,通过重定向url跳转网页
        # 第一个参数是我们想要跳转到的视图的名字name,(而不是写死的url),第二个参数是给该url指向的views方法传递的参数
        return HttpResponseRedirect(reverse('polls:results', args=(question_id,)))

