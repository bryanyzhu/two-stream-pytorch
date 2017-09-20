import ConfigParser

class mdp_config(object):
  config = ConfigParser.RawConfigParser()
  config.read('utils/util.config')

  