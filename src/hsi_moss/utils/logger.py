import logging

# Get the top-level logger object
log = logging.getLogger()

# make it print to the console.
console = logging.StreamHandler()
log.addHandler(console)
