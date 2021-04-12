from sys import stdout


def print_peforming_task(task: str):
    print_big("Performing %s ..." % (task))


def print_taks_done(task: str):
    print_big("%s Done!" % (task))


def print_big(string: str):
    stdout.write(
        "\n" +
        "=================================================" + "\n" +
        "| %s " % (string) + "\n"
        "=================================================" + "\n"
    )
    stdout.flush()


def print_percentages(prefix: str, percentage: float, icon: str = "="):
    stdout.write("%s [%-20s] %d%%" %
                 (prefix, icon*int(20*percentage), percentage*100))


def replace_print_flush(string: str):
    stdout.write("\r")
    stdout.write(string)
    stdout.flush()
