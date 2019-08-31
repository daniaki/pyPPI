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

    @classmethod
    def all(cls) -> peewee.ModelSelect:
        return cls.select()

    @classmethod
    def count(cls) -> int:
        return cls.all().count()

    def refresh(self):
        """Refresh from database"""
        return self.__class__.get(self._pk_expr())

    def format_for_save(self):
        """
        This method should be called prior to save to format fields. Always
        call super in child classes. Always call this in `save` method.
        """
        if self.created is None:
            self.created = datetime.datetime.now()
        self.modified = datetime.datetime.now()
        return self

    def save(self, *args, **kwargs):
        self.format_for_save()
        return super().save(*args, **kwargs)
