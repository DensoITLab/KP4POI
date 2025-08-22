class city:
    def __init__(self,name=''):
        self.name = name
        self.country = ''
        self.country_code = ''
        self.city_type = ''
    
    def get_name(self):
        return self.name

    def get_country(self):
        return (self.country,self.country_code)
    
    def get_type(self):
        return self.city_type
