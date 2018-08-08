from app.models import User
from app import db

users = User.query.all()
for u in users:
    db.session.delete(u)
db.session.commit()

user1 = User(username='Elena.Nesterova')
user1.set_password('InFhW6LK')
db.session.add(user1)

user2 = User(username='Olga.Ananieva')
user2.set_password('7peStJNq')
db.session.add(user2)

user3 = User(username='Marianna.Nurmukhametova')
user3.set_password('hiRaNyf9')
db.session.add(user3)

db.session.commit()