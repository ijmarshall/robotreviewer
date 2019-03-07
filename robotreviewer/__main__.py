import sys



if __name__ == '__main__':

    if len(sys.argv)==1 or "--rest" not in sys.argv[1:]:
        from robotreviewer import app
        if app.DEBUG_MODE:
            app.app.run(debug=True, use_reloader=False)
        else:
            app.app.run()
    else:
        from robotreviewer import cnxapp

        import connexion
        app = connexion.App(__name__, specification_dir='api/')
        app.add_api('robotreviewer_api.yml') 
        app.run(port=5000)




