// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_LABEL_P_H
#define _GTKMM_LABEL_P_H


#include <gtkmm/private/misc_p.h>
#include <gtk/gtk.h>

#include <glibmm/class.h>

namespace Gtk
{

class Label_Class : public Glib::Class
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef Label CppObjectType;
  typedef GtkLabel BaseObjectType;
  typedef GtkLabelClass BaseClassType;
  typedef Gtk::Misc_Class CppClassParent;
  typedef GtkMiscClass BaseClassParent;

  friend class Label;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  const Glib::Class& init();


  static void class_init_function(void* g_class, void* class_data);

  static Glib::ObjectBase* wrap_new(GObject*);

protected:

  //Callbacks (default signal handlers):
  //These will call the *_impl member methods, which will then call the existing default signal callbacks, if any.
  //You could prevent the original default signal handlers being called by overriding the *_impl method.
  static void populate_popup_callback(GtkLabel* self, GtkMenu* p0);

  //Callbacks (virtual functions):
};


} // namespace Gtk


#endif /* _GTKMM_LABEL_P_H */

