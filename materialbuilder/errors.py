class Error(Exception):
   pass

class TemplateError(Error):
   pass

class NodeTemplateError(Error):
   pass

class EdgeTemplateError(Error):
   pass

class PropTemplateError(Error):
   pass

class MissingAtomTypeError(Error):
    pass