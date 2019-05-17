import sys
from robotreviewer import config



if __name__ == '__main__':
    if config.REST_API == False:
        print("RUNNING WEB VERSION")
        from robotreviewer import app
        if app.DEBUG_MODE:
            app.app.run(debug=True, use_reloader=False)
        else:
            app.app.run()
    else:
        print("RUNNING REST API")
        from robotreviewer import cnxapp
        cnxapp.app.run()




