from datetime import datetime

date=datetime(year=2019,month=12,day=5,hour=20,minute=23)
print(date)

date.strftime('%A')
date.strftime('%B')
date.strftime('%C')
date.strftime('%D')
date.strftime('%Y')
date.strftime('%M')