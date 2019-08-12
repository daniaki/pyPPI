import datetime

import peewee

from ...settings import DATABASE


__all__ = ["BaseModel"]


class BaseModel(peewee.Model):
    FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    created = peewee.DateTimeField()
    modified = peewee.DateTimeField()

    class Meta:
        database = DATABASE

    def refresh(self):
        """Refresh from database"""
        return self.__class__.get(self._pk_expr())

    def save(self, *args, **kwargs):
        if self.created is None:
            self.created = datetime.datetime.now()
        self.modified = datetime.datetime.now()
        return super().save(*args, **kwargs)
