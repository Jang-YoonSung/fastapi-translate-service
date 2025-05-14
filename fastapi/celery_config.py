from celery import Celery
import os

import logging

rabbitmq_user = os.getenv('RABBITMQ_USER')
rabbitmq_pass = os.getenv('RABBITMQ_PASS')

logging.basicConfig(level=logging.INFO)
logging.info(f"RabbitMQ User: {rabbitmq_user}, RabbitMQ Pass: {rabbitmq_pass}, amqp://{rabbitmq_user}:{rabbitmq_pass}@translate_rabbitmq:5672")

celery_app = Celery(
    "worker",
    broker=f"amqp://{rabbitmq_user}:{rabbitmq_pass}@translate_rabbitmq:5672",
    backend=f"redis://translate_redis:6379/0",
)

celery_app.conf.update(
    task_track_started=True,
)