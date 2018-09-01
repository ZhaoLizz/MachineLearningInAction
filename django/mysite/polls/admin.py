from django.contrib import admin
from .models import Question, Choice


# Choice 对象要在 Question 后台页面编辑。默认提供 3 个选项字段。
class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 3


class QuestionAdmin(admin.ModelAdmin):
    """
    自定义后台表单要显示的数据
    """
    # fields = ['pub_date','question_text']
    # 字段集的显示方式
    fieldsets = [
        (None, {'fields': ['question_text']}),
        ('Date information', {'fields': ['pub_date']}),
    ]
    inlines = [ChoiceInline]
    list_display = ('question_text', 'pub_date', 'was_published_recently')
    list_filter = ['pub_date']
    search_fields = ['question_text']   # 搜索的字段


admin.site.register(Question, QuestionAdmin)
