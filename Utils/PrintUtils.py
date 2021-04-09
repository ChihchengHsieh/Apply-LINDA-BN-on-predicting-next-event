from sys import stdout

def printProgress(percentage, text):
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()

def printPerformedTask(text):
    stdout.write("\r%s" % text + "...      ")
    stdout.flush()

def printDoneTask():
    stdout.write("[DONE]")
    stdout.flush()
    stdout.write("\n")