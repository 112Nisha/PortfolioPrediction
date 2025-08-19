
from app import app
from flask_frozen import Freezer

freezer = Freezer(app)

# Register the endpoint name for freezing
@freezer.register_generator
def url_generator():
    yield '/'

if __name__ == '__main__':
    print('Freezing site...')
    freezer.freeze()
    print('Site frozen!')
