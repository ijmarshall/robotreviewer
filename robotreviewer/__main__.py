from robotreviewer import app

if __name__ == '__main__':
    if app.DEBUG_MODE:
        app.app.run(debug=True, use_reloader=False)
    else:
        app.app.run()

