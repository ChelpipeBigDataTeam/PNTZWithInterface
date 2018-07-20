from app.models import User
from app import db

user1 = User(username='Ivan Sokolov')
user1.set_password('sunset')
db.session.add(user1)

user2 = User(username='Alla Petrova')
user2.set_password('qween')
db.session.add(user2)

db.session.commit()