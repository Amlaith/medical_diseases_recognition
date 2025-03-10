## Распознавание пневмонии на рентгенограмме органов грудной клетки

Цель проекта - разработать сервис для автоматической идентификации пневмонии на основе анализа рентгенограммы.

Пройденые этапы: [checkpoints.md](https://github.com/Amlaith/medical_diseases_recognition/blob/main/checkpoint_1_meeting_the_team/checkpoint.md)

#### MVP
Иллюстрация работы FastAPI

(видео)

Иллюстрация работы Streamlit

(видео)

#### Подробнее о задаче
Сначала реализуется классификация по трём возможным исходам:
|Класс|Описание|
|:-|:-|
|Normal|без патологий|
|No Lung Opacity / Not Normal|нет затемнений характерных для пневмонии, но могут присутствовать затемнения другого генезиса|
|Lung Opacity|обнаружены признаки пневмонии|

Затем детектирование bounding box-ов в случае Lung Opacity.


#### Куратор проекта
Булыгин Глеб

Telegram: [@jdbelg](https://t.me/jdbelg)

#### Участники
|Участники|Telegram|Github|
|:-|:-|:-|
|Грицик Мария|[@Maria22032006](https://t.me/Maria22032006)|MariaGritsik|
|Заваруев Иван|[@Amlaith](https://t.me/Amlaith)|Amlaith|
|Терентьева Анастасия|[@manuls_are_cool](https://t.me/manuls_are_cool)|LovesManuls|


#### Организатор
Магистерская программа [НИУ ВШЭ "Искусственный интеллект"](https://www.hse.ru/ma/mlds/ "Страница программы на hse.ru")
