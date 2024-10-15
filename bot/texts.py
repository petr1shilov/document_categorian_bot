hello_message_text = 'Привет, я Умный бот 🤖'
start_message_text = ('Я создан, что бы определять категории в научных статей.\n'
                      'Работать со мной очень просто!\n'
                      'Просто отправьте мне файл с научной работой в формате pdf, а дальше я сделаю все сам!\n')

pdf_message_text = 'Отправьте, пожалуйста, файл pdf'

warning_pdf_message = ('К сожалению, это не pdf ☹️\n'
                       'Отправьте, пожалуйста, файл в формате pdf')

waiting_message = 'Идет фильтрация. Пожалуйста, подождите'

anchor_list = ["Обращено-фазная ВЭЖХ   с диодно-матричным или спектрофотометрическим детектором в ультрафиолетовой области",
               "Хроматограф высокоэффективный жидкостной включающий (насос с вакуумным дегазатором; термостат колонок; детектор диодно-матричный или спектрофотометрический (диапазон длин волн от 190 нм); автосамплер; система управления и обработки данных; колонка аналитическая хроматографическая (Пример: LichroCART 250-4 Superspher 100 RP-18); Весы лабораторные с диапазоном взвешивания 0,01-220 г, погрешностью 0,0002 г. Весы лабораторные с диапазоном взвешивания 0,5-510 г, погрешностью взвешивания 0,01 мг. Колбы мерные 2-50-2, 2-100-2, 2-250-2, 2-500-2, 2-1000-2 по ГОСТ 1770. Колба коническая КН-1-250-29/32 по ГОСТ 25336. Пипетка с одной меткой 2-2-5 по ГОСТ 29169. Пипетки градуированные 1-2-1-1, 1-2-1-5 по ГОСТ 29227. Стакан В-1-1000 ТС по ГОСТ 25336. Цилиндр 1-1000-1 по ГОСТ 1770. Баня ультразвуковая с частотой ультразвука 35 kHz. Виалы для автосамплера вместимостью 2 см. Термостат водяной с точностью поддержания температуры ±0,1°С, в диапазоне температур от 10°С до 40°С. Гомогенизатор для работы с объемом до 200 см или бытовой измельчитель. Картриджи для твердофазной экстракции вместимостью 3 см с привитой фазой С18 или C18-CN. Шейкер лабораторный с орбитальным или горизонтальным движением и регулируемой скоростью до 500 оборотов в минуту. Устройство для твердофазной экстракции. Устройство для фильтрования элюентов. Фильтры мембранные из регенерированной целлюлозы d=25 мм, =0,25 мкм. Фильтры мембранные из регенерированной целлюлозы d=47 мм, =0,45 мкм. Центрифуга, обеспечивающая скорость вращения 8000 оборотов в минуту с объемом стаканов для центрифугирования не менее V=30 см.Шприцы инъекционные однократного применения вместимостью 2; 10 см pH-метр с диапазоном измерений от 0 до 14 ед.pH и погрешностью 0,01 ед.pH в комплекте с электродами.",
               "Кислота фумаровая, массовая доля основного вещества не менее 99,0%. Кислота аскорбиновая, массовая доля основного вещества не менее 99,0%. Кислота винная, массовая доля основного вещества не менее 99,0%. Вода дистиллированная по ГОСТ 6709. Калий однозамещенный фосфорнокислый, содержание основного вещества не менее 99,7%. Кислота метафосфорная по ГОСТ 841, х.ч. Кислота ортофосфорная по ГОСТ 6552, х.ч. Кислота лимонная, массовая доля основного вещества не менее 99,0%. Кислота малеиновая, массовая доля основного вещества не менее 99,0%. Кислота молочная, массовая доля основного вещества не менее 88,0%. Спирт этиловый из пищевого сырья высшей степени очистки. Кислота уксусная, массовая доля основного вещества не менее 99,0%. Кислота шикимовая, массовая доля основного вещества не менее 98,0%. Кислота щавелевая, массовая доля основного вещества не менее 96,0%. Кислота яблочная, массовая доля основного вещества не менее 98,0%.Кислота янтарная, массовая доля основного вещества не менее 99,0%.",
               "Отбор пробы. Распространяется на напитки безалкогольные, слабоалкогольные, винодельческую продукцию, соки, нектары, сокосодержащие напитки, продукты переработки фруктов, ягод и овощей. Отбор проб продукции для определения содержания органических кислот должен обеспечивать однородность и репрезентативность представленной пробы, а также предусматривать отбор проб на случай разногласий в оценке качества продукции. Для выполнения анализа кислот  готовят две параллельные пробы. Каждую параллельную пробу перед измерением разбавляют в 2 и 50 раз для выполнения измерений массовых концентраций кислот в диапазоне градуировочного графика. От каждой параллельной пробы пипеткой отбирают 1 и 25 см3, переносят в две мерные колбы вместимостью 50 см3. Каждую колбу доводят до метки элюирующим раствором и хорошо перемешивают. Аликвоту объемом 5 см 3 из каждой колбы подготавливают далее в соответствии с требованиями",
               "Температура окружающего воздуха: 18°С-23°С; относительная влажность воздуха: 25 %-75 %; приготовление градуировочных растворов и растворов проб проводят при температуре: (20±2)°С.",
               "Картридж устанавливают в устройство для твердофазной экстракции (ТФЭ) и промывают последовательно 3 см3 этилового спирта и 4 см3 дистиллированной воды, не допуская осушения его рабочей поверхности. Из пробы отбирают аликвоту или супернатант объемом 5 см3 и пропускают через активированный картридж для ТФЭ при скорости фильтрования одна-две капли в секунду. Первую порцию фильтрата объемом 2 см3 отбрасывают. Последующую порцию фильтрата объемом около 3 см3 собирают и фильтруют в виалу для автосамплера через фильтр из регенерированной целлюлозы с диаметром пор 0,25 мкм. Подготовленная для измерения проба должна быть прозрачной. Приготовление элюирующего раствора: Навеску однозамещенного фосфорнокислого калия массой 2,72 г, взятую с точностью 0,01 г, переносят в мерную колбу вместимостью 1000 см3, заполненную 850-900 см3 дистиллированной водой, хорошо перемешивают. Значение pH приготовленного элюирующего раствора должно находиться в диапазоне 2,0-2,1 ед.pH. Значение pH регулируют с помощью 50%-ного раствора ортофосфорной кислоты. Раствор доводят до метки дистиллированной водой при температуре (20±2)°С. Перед применением элюирующий раствор фильтруют через фильтр из регенерированной целлюлозы с диаметром 47 мм, диаметром пор 0,45 мкм. Приготовление экстрагирующего раствора: 3%-ный раствор метафосфорной кислоты готовят в день проведения анализа. Навеску метафосфорной кислоты массой 30,0 г, взятую с точностью до 0,1 г, переносят в стакан вместимостью 1000 см3 и добавляют 970 см3 дистиллированной воды, отмеренной цилиндром вместимостью 1000 см3 при температуре (20±2)°С. Раствор хорошо перемешивают до полного растворения кислоты.",
               "Навески фумаровой и малеиновой кислот 0,125 г  с точностью до 0,001 г количественно переносят в мерную колбу вместимостью 500 см3, добавляют 450-460 см3 элюирующего раствора,  хорошо перемешивают до полного растворения и доводят до метки элюирующим раствором. Навески индивидуальных органических кислот  взвешивают с точностью до 0,001 г в мерной колбе вместимостью 250 см3, предварительно заполненной 40-50 см3 элюирующего раствора. Добавляют 150 см3 элюирующего раствора Раствор хорошо перемешивают до полного растворения кислот. Затем в колбу добавляют 25 см3 основного градуировочного раствора фумаровой и малеиновой кислот и доводят до метки элюирующим раствором. Используют 0,250 г винной, яблочной кислот ( массовая концентрация 1000 мг/дм3), 0,312 г щавеливой кислоты (1250 мг/дм3), 1,875 г молочной кислоты (7500 мг/дм3), 1,250г уксусной, янтарной кислот (5000 мг/дм3), 0,625 г лимонной кислоты (2500 мг/дм3), 0,125 г шикимовой кислоты (500 мг/дм3). Рабочие градуировочные растворы готовят путем разбавления основного градуировочного раствора смеси органических кислот: N 1- Элюирующий раствор; N 2 готовят из градуировочного раствора N 7: Объем растора 5 см3, в колбе вместимостью 100 см3; N 3 - Объем основного градуировочного растора 1 см3, в колбе вместимостью 100 см3; N 4 - Объем основного градуировочного растора 2,5 см3, в колбе вместимостью 100 см3; N 5 - Объем основного градуировочного растора 2,5 см3, в колбе вместимостью 50 см3; N 6 - Объем основного градуировочного растора 4 см3, в колбе вместимостью 50 см3.  Для определения рабочего диапазона массовых концентраций и линейности функции градуировки выполняют градуировку хроматографа по семи рабочим градуировочным растворам. Градуировочная зависимость должна иметь вид y=bx+а. Параметры градуировочной зависимости вычисляют путем анализа линейной регрессии скорректированных площадей пиков, учитывая нулевую точку",
               "Колонка аналитическая, обращенно-фазная длина 250 мм, внутренний диаметр 4,6 мм, заполненная сорбентом, с размером частиц 5 мкм. Элюент: КН2РО4 молярной концентрации 0,02 моль/дм, pH раствора 2,0-2,1 ед.pH. Режим элюирования: изократический. Температура колонки: (25,0±0,1)°С. Скорость подачи элюента: 0,4 см/мин.  Детектор: спектрофотометрический или диодная матрица. Длина волны поглощения: 220 нм. Время анализа: 25 мин. Объем вводимой пробы: 20-50 мм.",
               "Пробу хроматографируют 2 раза. Пики кислот в пробе должны быть правильно проинтегрированы, при необходимости результаты интегрирования площади пика корректируют вручную. Если измеренная массовая концентрация кислоты в анализируемом растворе превышает верхнюю границу диапазона градуировки",
               "При отсутствии пика анализируемой кислоты на характеристическом времени удерживания, соответствующем времени, полученном на хроматограмме градуировочного раствора, делают вывод, что данная кислота в пробе отсутствует. Если время удерживания определяемой кислоты на хроматограмме анализируемой пробы совпадает со временем удерживания анализируемой кислоты на хроматограмме градуировочного раствора, то считают, что данная кислота в пробе присутствует. Массовую концентрацию Сi1-ой кислоты в параллельной пробе вычисляют по формуле - для жидких проб в мг/дм: Ci1 =Cxi*K, где Cxi- массовая концентрация i-ой кислоты в анализируемом растворе, вычисленная по градуировочному графику, мг/дм; K- коэффициент разбавления, вычисляемыq по формулам K=Vp/Va, где Vp- объем разбавленного раствора пробы, см (50 см3), Va- объем исходной пробы, см3 (1 см3 или 25 см3). Вычисление результата измерений выполняют отдельно для каждого разбавления. Вычисляют среднеарифметическое значение результатов измерений двух параллельных проб. Результат измерений при доверительной вероятности Р=0,95 представляют в следующем виде ( Сср ±Абсолютное значение показателя точности )",
               "Ориентировочное время удерживания щавелевой кислоты: 5,5  мин Диапазон измерения массовой концентрации щавелевой кислоты: 10-1500 мг/дм3",
               "Необходимость построения градуировочной зависимости. Длительная пробоподготовка. Идентификация. Количественное определение.  Точность:  массовая концентрация от 10 до 250 включ. г/дм3 с погрешностью 25,57% ;  от  250 до 1500 включ. г/дм3 с погрешностью 13,36%"]

system_prompt = 'Тебе необходимо определить подходит ли текст под категорию или нет\n'

column_name_list = ["Метод исследования",	"Приборы и оборудование",
                              	"Реактивы и материалы",	"Отбор пробы",	"Условия проведения измерений",
                              	"Подготовка к работе", "Подготовка градуировочной зависимости", 	
                                "Режим работы прибора", "Проведение анализа",	"Обработка результатов",
                              	"Результат", "Оценка используемого метода"]

discription_prompt_list = [
    ('''
    "properties": {
    "Метод исследования": {       
      "type": "string",       
      "description": "Какие методы исполюзуются в исследованиях"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Метод исследования" : "Обращено-фазная ВЭЖХ с диодно-матричным или спектрофотометрическим детектором в ультрафиолетовой области "
'''),
('''
    "properties": {
    ""Приборы и оборудование"": {       
      "type": "string",       
      "description": "Какое оборудование или приборы используются в исследовании"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  ""Приборы и оборудование"" : ["Хроматограф высокоэффективный жидкостной включающий (насос с вакуумным дегазатором; термостат колонок; детектор диодно-матричный или спектрофотометрический (диапазон длин волн от 190 нм); автосамплер; система управления и обработки данных; колонка аналитическая хроматографическая (Пример: LichroCART 250-4 Superspher 100 RP-18); Весы лабораторные с диапазоном взвешивания 0,01-220 г, погрешностью 0,0002 г. Весы лабораторные с диапазоном взвешивания 0,5-510 г, погрешностью взвешивания 0,01 мг. Колбы мерные 2-50-2, 2-100-2, 2-250-2, 2-500-2, 2-1000-2 по ГОСТ 1770. Колба коническая КН-1-250-29/32 по ГОСТ 25336. Пипетка с одной меткой 2-2-5 по ГОСТ 29169. Пипетки градуированные 1-2-1-1, 1-2-1-5 по ГОСТ 29227. Стакан В-1-1000 ТС по ГОСТ 25336. Цилиндр 1-1000-1 по ГОСТ 1770. Баня ультразвуковая с частотой ультразвука 35 kHz. Виалы для автосамплера вместимостью 2 см. Термостат водяной с точностью поддержания температуры ±0,1°С, в диапазоне температур от 10°С до 40°С. Гомогенизатор для работы с объемом до 200 см или бытовой измельчитель. Картриджи для твердофазной экстракции вместимостью 3 см с привитой фазой С18 или C18-CN. Шейкер лабораторный с орбитальным или горизонтальным движением и регулируемой скоростью до 500 оборотов в минуту. Устройство для твердофазной экстракции. Устройство для фильтрования элюентов. Фильтры мембранные из регенерированной целлюлозы d=25 мм, =0,25 мкм. Фильтры мембранные из регенерированной целлюлозы d=47 мм, =0,45 мкм. Центрифуга, обеспечивающая скорость вращения 8000 оборотов в минуту с объемом стаканов для центрифугирования не менее V=30 см.
Шприцы инъекционные однократного применения вместимостью 2; 10 см 
pH-метр с диапазоном измерений от 0 до 14 ед.pH и погрешностью 0,01 ед.pH в комплекте с электродами.",
"Титрометрия"]
'''),
('''
    "properties": {
    "Реактивы и материалы": {       
      "type": "string",       
      "description": "Какие материалы или реактивы использубтся в исследовании"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Реактивы и материалы" : "Кислота фумаровая, массовая доля основного вещества не менее 99,0%. Кислота аскорбиновая, массовая доля основного вещества не менее 99,0%.
Кислота винная, массовая доля основного вещества не менее 99,0%.
Вода дистиллированная по ГОСТ 6709.
Калий однозамещенный фосфорнокислый, содержание основного вещества не менее 99,7%.
Кислота метафосфорная по ГОСТ 841, х.ч.
Кислота ортофосфорная по ГОСТ 6552, х.ч.
Кислота лимонная, массовая доля основного вещества не менее 99,0%.
Кислота малеиновая, массовая доля основного вещества не менее 99,0%.
Кислота молочная, массовая доля основного вещества не менее 88,0%.
Спирт этиловый из пищевого сырья высшей степени очистки.
Кислота уксусная, массовая доля основного вещества не менее 99,0%.
Кислота шикимовая, массовая доля основного вещества не менее 98,0%.
Кислота щавелевая, массовая доля основного вещества не менее 96,0%.
Кислота яблочная, массовая доля основного вещества не менее 98,0%.
Кислота янтарная, массовая доля основного вещества не менее 99,0%."
'''),
('''
    "properties": {
    "Отбор пробы": {       
      "type": "string",       
      "description": "Откуда берется образец пробы"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Отбор пробы" : [" Распространяется на напитки безалкогольные, слабоалкогольные, винодельческую продукцию, соки, нектары, сокосодержащие напитки, продукты переработки фруктов, ягод и овощей. Отбор проб продукции для определения содержания органических кислот должен обеспечивать однородность и репрезентативность представленной пробы, а также предусматривать отбор проб на случай разногласий в оценке качества продукции. Для выполнения анализа кислот  готовят две параллельные пробы. Каждую параллельную пробу перед измерением разбавляют в 2 и 50 раз для выполнения измерений массовых концентраций кислот в диапазоне градуировочного графика.
От каждой параллельной пробы пипеткой отбирают 1 и 25 см3, переносят в две мерные колбы вместимостью 50 см3. Каждую колбу доводят до метки элюирующим раствором и хорошо перемешивают. Аликвоту объемом 5 см 3 из каждой колбы подготавливают далее в соответствии с требованиями.",
"Подготовка пробы. Берут навеску массой 1 г, взвешенную с записью до третьего десятичного знака, помещают в виалу вместимостью 30 см3, добавляют 25 см3 дистиллированной воды и взбалтывают до получения однородной суспензии. Плотно закрывают крышкой и помещают в ультразвуковую баню. Проводят экстракцию органических кислот на ультразвуковой бане. Из средней пробы измельченного изделия по разделу 8 берут навеску массой 1 г. взвешенную с 
записью до третьего десятичного знака, помещают в виалу вместимостью 30 см3, добавляют 25 см3 дис­
тиллированной воды и взбалтывают до получения однородной суспензии. Плотно закрывают крышкой и помещают в ультразвуковую баню.",
"Воздух с объемным расходом 10 дм/мин аспирируют через фильтр АФА-ХА-20, помещенный в фильтродержатель. Для измерения ПДК щавелевой кислоты дигидрата необходимо отобрать 120 дм3 воздуха."]'''),
('''
    "properties": {
    "Условия проведения измерений": {       
      "type": "string",       
      "description": "Информация о параметрах внешней среды и условиях, необходимых для проведения измерений и экспериментов. Включает диапазон температуры и влажности окружающего воздуха, а также специальные требования к температурным условиям при приготовлении растворов и проведении анализов."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Условия проведения измерений" : "Температура окружающего воздуха: 18°С-23°С; относительная влажность воздуха: 25%-75; приготовление градуировочных растворов и растворов проб проводят при температуре: (20±2)°С. ",
  '''),

('''
    "properties": {
    "Подготовка к работе": {       
      "type": "string",       
      "description": "Описываются процедуры и этапы, необходимые для подготовки оборудования, материалов и реактивов перед проведением эксперимента или анализа"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Подготовка к работе" : "Картридж устанавливают в устройство для твердофазной экстракции (ТФЭ) и промывают последовательно 3 см3 этилового спирта и 4 см3 дистиллированной воды, не допуская осушения его рабочей поверхности. Из пробы отбирают аликвоту или супернатант объемом 5 см3 и пропускают через активированный картридж для ТФЭ при скорости фильтрования одна-две капли в секунду. Первую порцию фильтрата объемом 2 см3 отбрасывают. Последующую порцию фильтрата объемом около 3 см3 собирают и фильтруют в виалу для автосамплера через фильтр из регенерированной целлюлозы с диаметром пор 0,25 мкм. Подготовленная для измерения проба должна быть прозрачной. Приготовление элюирующего раствора: Навеску однозамещенного фосфорнокислого калия массой 2,72 г, взятую с точностью 0,01 г, переносят в мерную колбу вместимостью 1000 см3, заполненную 850-900 см3 дистиллированной водой, хорошо перемешивают. Значение pH приготовленного элюирующего раствора должно находиться в диапазоне 2,0-2,1 ед.pH. Значение pH регулируют с помощью 50%-ного раствора ортофосфорной кислоты. Раствор доводят до метки дистиллированной водой при температуре (20±2)°С. Перед применением элюирующий раствор фильтруют через фильтр из регенерированной целлюлозы с диаметром 47 мм, диаметром пор 0,45 мкм. Приготовление экстрагирующего раствора: 3%-ный раствор метафосфорной кислоты готовят в день проведения анализа 
Навеску метафосфорной кислоты массой 30,0 г, взятую с точностью до 0,1 г, переносят в стакан вместимостью 1000 см3 и добавляют 970 см3 дистиллированной воды, отмеренной цилиндром вместимостью 1000 см3 при температуре (20±2)°С. Раствор хорошо перемешивают до полного растворения кислоты.", 
'''),
('''
    "properties": {
    "Подготовка градуировочной зависимости": {       
      "type": "string",       
      "description": "подробный процесс подготовки градуировочной зависимости для различных органических кислот. Описаны процедуры взвешивания навесок кислот с точностью до 0,001 г, их растворения в мерных колбах определенного объема с использованием элюирующего раствора, а также порядок добавления основных и рабочих градуировочных растворов. Осуществляется расчет концентраций каждой из кислот в рабочих растворах и их разведение до нужного объема."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Подготовка градуировочной зависимости" : ["Навески фумаровой и малеиновой кислот 0,125 г  с точностью до 0,001 г количественно переносят в мерную колбу вместимостью 500 см3, добавляют 450-460 см3 элюирующего раствора,  хорошо перемешивают до полного растворения и доводят до метки элюирующим раствором. Навески индивидуальных органических кислот  взвешивают с точностью до 0,001 г в мерной колбе вместимостью 250 см3, предварительно заполненной 40-50 см3 элюирующего раствора. Добавляют 150 см3 элюирующего раствора Раствор хорошо перемешивают до полного растворения кислот. Затем в колбу добавляют 25 см3 основного градуировочного раствора фумаровой и малеиновой кислот и доводят до метки элюирующим раствором. Используют 0,250 г винной, яблочной кислот ( массовая концентрация 1000 мг/дм3), 0,312 г щавеливой кислоты (1250 мг/дм3), 1,875 г молочной кислоты (7500 мг/дм3), 1,250г уксусной, янтарной кислот (5000 мг/дм3), 0,625 г лимонной кислоты (2500 мг/дм3), 0,125 г шикимовой кислоты (500 мг/дм3). Рабочие градуировочные растворы готовят путем разбавления основного градуировочного раствора смеси органических кислот: N 1- Элюирующий раствор; N 2 готовят из градуировочного раствора N 7: Объем растора 5 см3, в колбе вместимостью 100 см3; N 3 - Объем основного градуировочного растора 1 см3, в колбе вместимостью 100 см3; N 4 - Объем основного градуировочного растора 2,5 см3, в колбе вместимостью 100 см3; N 5 - Объем основного градуировочного растора 2,5 см3, в колбе вместимостью 50 см3; N 6 - Объем основного градуировочного растора 4 см3, в колбе вместимостью 50 см3.  Для определения рабочего диапазона массовых концентраций и линейности функции градуировки выполняют градуировку хроматографа по семи рабочим градуировочным растворам. Градуировочная зависимость должна иметь вид y=bx+а. Параметры градуировочной зависимости вычисляют путем анализа линейной регрессии скорректированных площадей пиков, учитывая нулевую точку.", 
  "Градуировочный раствор смеси органических кислот массовой концентрацией каждой кислоты 2 мг/дм3 В виалу вместимостью 4 см3 с помощью пипеточного дозатора вносят по 60 мм3 основных растворов щавелево, винной, яблочной и лимонной кислот и 2760 мм3 дистиллированной воды. Раствор готовят перед использованием. Градуировочный раствор смеси органических кислот массовой концентрацией каждой кислоты 5 мг/дм3 В виалу вместимостью 4 см3 с помощью пипеточного дозатора вносят по 150 мм3 основных растворов щавелевой, винной, яблочной  и лимонной кислот и 2400 мм3 дистиллированной воды. Раствор готовят перед использованием. Градуировочный раствор смеси органических кислот массовой концентрацией каждой кислоты 10 мг/дм3 В виалу вместимостью 4 см3 с помощью пипеточного дозатора вносят по 300 мм3 основных растворов щавелевой, винной, яблочной и лимонной кислот и 1800 мм3 дистиллированной воды.",
  "Приготовление градуировочных растворов органических кислот Градуировочный раствор смеси органических кислот массовой концентрацией каждой кислоты 2 мг/дм3В еиалу вместимостью 4 см3 с помощью лилеточного дозатора вносят по 60 мм3 основных раство­ров щавелевой (см. 9.4.1). винной (см. 9.4.2). яблочной (см. 9.4.3) илимонной (см. 9.4.4)кислоти2760 мм3 дистиллированной воды.Раствор готовят перед использованием.Градуировочный раствор смеси органических кислот массовой концентрацией каждой кислоты 5 мг/дм3",
  "Градуировочную характеристику, выражающую зависимость оптической плотности растворов от массы щавелевой кислоты дигидрата, устанавливают по 6 сериям растворов из 5 параллельных определений для каждой серии. Стандарт 1: Стандартный раствор щавелевой кислоты дигидрат N 2: 0 см3; Метанол: 2,0 см3; Содержание щавелевой кислоты дигидрата в градуировочном растворе: 0 мкг. Стандарт 2: Стандартный раствор щавелевой кислоты дигидрат N 2: 0,1 см3"] 	
'''),
('''
    "properties": {
    "Режим работы прибора": {       
      "type": "string",       
      "description": "содержит информацию о параметрах работы аналитической колонки, используемой в хроматографическом анализе. Описание включает тип и характеристики колонки (например, аналитическая, обращенно-фазная), ее размеры (длина и внутренний диаметр), свойства сорбента (размер частиц, химический состав), а также параметры элюента (молярная концентрация, pH). Указываются режим элюирования (например, изократический), температура работы колонки, скорость подачи элюента, тип детектора, длина волны поглощения, время анализа и объем вводимой пробы."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Режим работы прибора" : ["Колонка аналитическая, обращенно-фазная длина 250 мм, внутренний диаметр 4,6 мм, заполненная сорбентом, с размером частиц 5 мкм.
Элюент: КН2РО4 молярной концентрации 0,02 моль/дм, pH раствора 2,0-2,1 ед.pH.
Режим элюирования: изократический.
Температура колонки: (25,0±0,1)°С.
Скорость подачи элюента: 0,4 см/мин.
 Детектор: спектрофотометрический или диодная матрица.
Длина волны поглощения: 220 нм.
Время анализа: 25 мин.
Объем вводимой пробы: 20-50 мм.",
"Капилляр с внутренним диаметром 75 мкм, общей длиной 60 см. Систему капиллярного электрофореза подготавливают к работе в соответствии с руководством (инструкцией) по эксплуатации и устанавливают следующие рабочие параметры:  
- рабочее напряжение  - минус 17 кВ;
- рекомендуемая температура для анализа - 25°С;
- длительность анализа - 15 мин;
- ввод пробы гидродинамический под давлением - 150 мбар9 с.",
"Подготовка системы капиллярного электрофореза к измерениям. Для детектирования органических кислот используют косвенный метод при длине волны 235 нм.  Капилляр свнутренним диаметром 75 мкм. общей длиной 60 см. Систему капиллярного электрофореза  подготавливают кработе в соответствии с руководством (инструкцией) по эксплуатации и устанавлива­ют следующие рабочие параметры:",
"Измерение производят при длине волны 490 нм."]
'''),
('''
    "properties": {
    "Проведение анализа": {       
      "type": "string",       
      "description": "Описание процесса проведения анализа и требований к интерпретации данных. Включает информацию о количестве повторов измерений, методах обработки данных, таких как интеграция пиков и корректировка результатов. Особое внимание уделяется проверке соответствия результатов установленным стандартам и диапазонам измерений, а также корректировке данных в случае превышения допустимых границ."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Проведение анализа" : ["Пробу хроматографируют 2 раза. Пики кислот в пробе должны быть правильно проинтегрированы, при необходимости результаты интегрирования площади пика корректируют вручную. Если измеренная массовая концентрация кислоты в анализируемом растворе превышает верхнюю границу диапазона градуировки.",
  "Подготовка капилляра к работе Перед измерением подготавливают капилляр к работе, промывая его раствором",
  "Подготовка нового капилляра Подготовку нового капилляра к работе проводят в соответствии с руководством (инструкцией) по  эксплуатации прибора. При отсутствии в руководстве (инструкции) указаний к подготовке нового капил­ ляра его последовательно промывают: дистиллированной водой — 10 мин. раствором 0,5 моль/дм3  соляной кислоты по 9.6 — 30 мин. дистиллированной водой — 10 мин, раствором 0.5 моль/дм3 гидроо­ киси натрия по 9.7 — 30 мин. дистиллированной водой —10 мин и буферным раствором по 9.3 — 30 мин. 10.2.2 Подготовка капилляра к работе Перед измерением подготавливают капилляр к работе, промывая его раствором 0.1 моль/дм3 гид­ роокиси натрия (см. 9.1) в течение 4 мин (давление ввода 1500 мбар), затем буферным раствором  (см. 9.3) в течение 6 мин (давление ввода 1500 мбар). Капилляр промывают каждый раз при включении прибора. 10.2.3 Промывка капилляра между измерениями и в конце рабочего дня Между измерениями капилляр промывают раствором 0.1 моль/дмэ гидроокиси натрия (см. 9.1) в  течение 4 мин (давление ввода 1500 мбар). затем буферным раствором (см. 9.3)в течение 6 мин (давле-  ние ввода 1500 мбар). После завершения измерений капилляр промывают в течение 10 мин раствором гидроокиси  натрия (см. 9.1) и оставляют концы капилляра погруженными в пробирки с дистиллированной водой. П р и м е ч а н и е  — При обнаружении не злекгрофореграммах дрейфа нулевой линии, ступеней и смеще­ ния времен миграции органических кислот рекомендуется. • увеличить продолжительность промывания капилляра буферным раствором между измерениями. • провести промывание капилляра буферным раствором при рабочем напряжении а течение 2—3 мин. • заменить буферный раствор в вивлах на входе и на выходе свежими порциями."]
  '''),
('''
    "properties": {
    "Обработка результатов": {       
      "type": "string",       
      "description": "содержит инструкции и алгоритмы для интерпретации данных и вычисления количественных характеристик анализируемых веществ. В этой колонке описываются методы сопоставления экспериментальных данных с эталонными значениями, критерии подтверждения наличия или отсутствия веществ в пробе, а также формулы для расчета концентраций и других показателей. Включаются условия и параметры расчета, такие как разбавление проб, объемы растворов и доверительные интервалы, а также процедуры усреднения и представления результатов измерений."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Обработка результатов" : "При отсутствии пика анализируемой кислоты на характеристическом времени удерживания, соответствующем времени, полученном на хроматограмме градуировочного раствора, делают вывод, что данная кислота в пробе отсутствует.
Если время удерживания определяемой кислоты на хроматограмме анализируемой пробы совпадает со временем удерживания анализируемой кислоты на хроматограмме градуировочного раствора, то считают, что данная кислота в пробе присутствует. Массовую концентрацию Сi1-ой кислоты в параллельной пробе вычисляют по формуле - для жидких проб в мг/дм: Ci1 =Cxi*K, где Cxi- массовая концентрация i-ой кислоты в анализируемом растворе, вычисленная по градуировочному графику, мг/дм; K- коэффициент разбавления, вычисляемыq по формулам K=Vp/Va, где Vp- объем разбавленного раствора пробы, см (50 см3), Va- объем исходной пробы, см3 (1 см3 или 25 см3). Вычисление результата измерений выполняют отдельно для каждого разбавления. Вычисляют среднеарифметическое значение результатов измерений двух параллельных проб. Результат измерений при доверительной вероятности Р=0,95 представляют в следующем виде ( Сср ±Абсолютное значение показателя точности )",
'''),
('''
    "properties": {
    "Результат": {       
      "type": "string",       
      "description": "Формирование результата после исследования"     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  "Результат" : "Ориентировочное время удерживания щавелевой кислоты: 5,5  мин Диапазон измерения массовой концентрации щавелевой кислоты: 10-1500 мг/дм3. Диапазон измерений массовой доли щавелевой кислоты", 
'''),
('''
    "properties": {
    "Оценка используемого метода": {       
      "type": "string",       
      "description": "Содержит подробную информацию о ключевых характеристиках и ограничениях применяемого метода анализа. Здесь указываются такие параметры, как необходимость в предварительной пробоподготовке и построении градуировочной зависимости, возможности идентификации и количественного определения веществ. Оценивается точность метода в зависимости от диапазона концентраций анализируемого компонента, с указанием допустимой погрешности для каждой из градаций."     
    }\n'''
"Выведи ответ в формате bool True или False\n"
'''Пример категории:
 {
  ["Оценка используемого метода" : "Необходимость построения градуировочной зависимости. Длительная пробоподготовка. Идентификация. Количественное определение.  Точность:  массовая концентрация от 10 до 250 включ. г/дм3 с погрешностью 25,57% ;  от  250 до 1500 включ. г/дм3 с погрешностью 13,36%.",
  "Необходимость построения градуировочной зависимости. Длительная пробоподготовка. Идентификация и количественное определение.Показатель точности: 14%",
  "Результат определения массовой доли органических кислот представляют в виде где Сср — среднеарифметическое значение результатов двух параллельных определений массовой  доли органических кислот. %; д — значение абсолютной погрешности определения массовой доли органических кислот. %. которое  рассчитывается по формуле Д= 8-0.01 Сср.  где 8— границы относительной погрешности (см. таблицу 1). %. Числовое значение результата измерений должно оканчиваться цифрой того же разряда, что и  значение абсолютной погрешности измерений, содержащее не более двух значащих цифр. Настоящий метод выполнения измерений обеспечивает получение результатов измерений в диа­ пазонах ис показателем точности и пределами повторяемости и воспроизводимости приР= 0.95.",
  "Не требует построения градуировочной зависимости. Количественное опредление. Влияние мешающих агентов"]
'''),
]