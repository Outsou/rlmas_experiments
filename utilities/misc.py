def agent_name_parse(name):
    parsed_name = name.replace('://', '_')
    parsed_name = parsed_name.replace(':', '_')
    parsed_name = parsed_name.replace('/', '_')
    return parsed_name